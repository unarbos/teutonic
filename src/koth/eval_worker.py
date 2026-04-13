#!/usr/bin/env python3
"""Self-contained eval worker that runs on an ephemeral Lium pod.

This script is SCP'd to a clean GPU machine and executed. It:
1. Downloads king and challenger models from HuggingFace (safetensors only)
2. Downloads an eval data shard from R2
3. Validates the challenger (architecture match, bounding box)
4. Runs the sequential sign test with early stopping
5. Streams per-sequence outcomes to R2
6. Writes the final verdict to R2

All results go to R2 -- the coordinator polls R2 to track progress.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import torch
from botocore.config import Config as BotoConfig
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
from scipy.stats import binom

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_worker")

# ---------------------------------------------------------------------------
# R2 helpers (inline so this script is fully self-contained)
# ---------------------------------------------------------------------------

class R2:
    def __init__(self, endpoint_url: str, bucket: str, key_id: str, secret: str):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            region_name="auto",
            config=BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"}),
        )

    def put_json(self, key: str, data) -> None:
        body = json.dumps(data, default=str).encode()
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json")

    def append_jsonl(self, key: str, record: dict) -> None:
        line = json.dumps(record, default=str) + "\n"
        existing = b""
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
            existing = resp["Body"].read()
        except Exception:
            pass
        self.client.put_object(
            Bucket=self.bucket, Key=key,
            Body=existing + line.encode(),
            ContentType="application/x-ndjson",
        )

    def download_file(self, key: str, local_path: str) -> None:
        self.client.download_file(self.bucket, key, local_path)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

SAFETENSORS_PATTERNS = ["*.safetensors"]
IGNORE_PATTERNS = ["*.bin", "*.pt", "*.pkl", "*.pickle", "*.py", "*.pyc", "*.sh", "__pycache__/*", ".git/*"]


def download_model(repo: str, target_dir: str, hf_token: str = "") -> Path:
    local = snapshot_download(
        repo_id=repo,
        local_dir=target_dir,
        token=hf_token or None,
        allow_patterns=SAFETENSORS_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )
    return Path(local)


def load_state_dict(directory: Path) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for f in sorted(directory.glob("*.safetensors")):
        sd.update(load_safetensors(str(f)))
    if not sd:
        raise FileNotFoundError(f"No .safetensors files in {directory}")
    return sd

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_challenger(king_sd, challenger_sd, cfg) -> tuple[bool, str, str]:
    """Returns (valid, reason, detail)."""
    king_keys = set(king_sd.keys())
    challenger_keys = set(challenger_sd.keys())

    if king_keys != challenger_keys:
        missing = king_keys - challenger_keys
        extra = challenger_keys - king_keys
        return False, "key_mismatch", f"missing={list(missing)[:3]}, extra={list(extra)[:3]}"

    for name in king_keys:
        if king_sd[name].shape != challenger_sd[name].shape:
            return False, "shape_mismatch", f"{name}: {king_sd[name].shape} vs {challenger_sd[name].shape}"
        if king_sd[name].dtype != challenger_sd[name].dtype:
            return False, "dtype_mismatch", f"{name}: {king_sd[name].dtype} vs {challenger_sd[name].dtype}"

    frozen = cfg.get("bbox_frozen_prefixes", [])
    max_linf = cfg.get("bbox_max_linf", 0.5)
    max_l2_per = cfg.get("bbox_max_l2_per_tensor")
    max_l2_global = cfg.get("bbox_max_l2_global")

    l2_sq_total = 0.0
    for name in king_keys:
        if any(name.startswith(p) for p in frozen):
            delta = (challenger_sd[name].float() - king_sd[name].float()).abs()
            if delta.max().item() > 1e-8:
                return False, "frozen_modified", name
            continue

        delta = challenger_sd[name].float() - king_sd[name].float()
        linf = delta.abs().max().item()
        if linf > max_linf:
            return False, "linf_violation", f"{name}: {linf:.6f} > {max_linf}"

        l2 = delta.norm(2).item()
        if max_l2_per is not None and l2 > max_l2_per:
            return False, "l2_tensor_violation", f"{name}: {l2:.4f} > {max_l2_per}"

        l2_sq_total += l2 ** 2

    if max_l2_global is not None and l2_sq_total ** 0.5 > max_l2_global:
        return False, "l2_global_violation", f"{l2_sq_total**0.5:.4f} > {max_l2_global}"

    return True, "", ""

# ---------------------------------------------------------------------------
# Model wrapper for eval
# ---------------------------------------------------------------------------

class ModelForEval:
    """Wraps a state dict into an nn.Module for forward-pass loss computation."""

    def __init__(self, state_dict: dict[str, torch.Tensor], device: str = "cuda"):
        self.device = device
        self.sd = {k: v.to(device) for k, v in state_dict.items()}

        # Detect model structure from parameter names to build the right model.
        # We use a generic transformer forward pass based on the state dict keys.
        self._model = self._build_model()
        self._model.eval()

    def _build_model(self) -> torch.nn.Module:
        """Build an nn.Module from the state dict.

        Attempts to use transformers.AutoModelForCausalLM if available,
        otherwise falls back to a manual weight-loading approach.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoConfig

            # Try to infer config from the state dict structure
            # For now we just load a generic causal LM
            config_keys = {k.split(".")[0] for k in self.sd.keys()}

            # Create a temporary dir with safetensors to load from
            import tempfile
            from safetensors.torch import save_file

            with tempfile.TemporaryDirectory() as tmpdir:
                save_file(self.sd, f"{tmpdir}/model.safetensors")
                # We need a config.json -- this won't work without one
                raise ImportError("Need config.json")

        except (ImportError, Exception):
            pass

        # Fallback: manual forward pass using raw tensors
        return _RawTransformerEval(self.sd, self.device)

    @torch.no_grad()
    def loss_on_tokens(self, tokens: torch.Tensor, use_amp: bool = True, amp_dtype: str = "bfloat16") -> float:
        dtype = getattr(torch, amp_dtype, torch.bfloat16) if use_amp else torch.float32
        tokens = tokens.to(self.device).unsqueeze(0) if tokens.dim() == 1 else tokens.to(self.device)
        with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
            return self._model.compute_loss(tokens)


class _RawTransformerEval(torch.nn.Module):
    """Generic causal LM forward pass from raw state dict tensors.

    Supports LLaMA-like architectures by detecting the parameter naming pattern
    and manually computing the forward pass. This avoids any dependency on
    transformers or custom model code.
    """

    def __init__(self, sd: dict[str, torch.Tensor], device: str):
        super().__init__()
        self.sd = sd
        self.device = device
        self._detect_architecture()

    def _detect_architecture(self) -> None:
        """Detect the model architecture from parameter name patterns."""
        keys = set(self.sd.keys())
        # Check for common LLaMA-like patterns
        self.has_lm_head = "lm_head.weight" in keys
        self.has_embed = "model.embed_tokens.weight" in keys or "embed_tokens.weight" in keys
        self.embed_key = "model.embed_tokens.weight" if "model.embed_tokens.weight" in keys else "embed_tokens.weight"

        # Count layers
        layer_indices = set()
        for k in keys:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_indices.add(int(parts[i + 1]))
        self.n_layers = max(layer_indices) + 1 if layer_indices else 0

    def compute_loss(self, tokens: torch.Tensor) -> float:
        """Compute cross-entropy loss on a token sequence.

        If we can't reconstruct the full forward pass from the state dict,
        we fall back to loading via a simpler approach.
        """
        if not self.has_embed or not self.has_lm_head:
            raise RuntimeError(
                "Cannot compute loss: model structure not recognized. "
                "Ensure the king repo includes a config.json so we can use AutoModelForCausalLM."
            )

        # For production, the eval worker should receive a config.json
        # and use AutoModelForCausalLM. This fallback exists for testing.
        raise NotImplementedError(
            "Raw forward pass not implemented. "
            "The eval pod should have a config.json in the king directory."
        )


class HFModelForEval:
    """Load and eval using HuggingFace transformers (preferred path)."""

    def __init__(self, model_dir: str | Path, device: str = "cuda"):
        from transformers import AutoModelForCausalLM

        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=False,
        )
        self.model.eval()

    @torch.no_grad()
    def loss_on_tokens(self, tokens: torch.Tensor, use_amp: bool = True, amp_dtype: str = "bfloat16") -> float:
        tokens = tokens.to(self.device).unsqueeze(0) if tokens.dim() == 1 else tokens.to(self.device)
        dtype = getattr(torch, amp_dtype, torch.bfloat16) if use_amp else torch.float32
        with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
            outputs = self.model(input_ids=tokens[:, :-1], labels=tokens[:, 1:])
        return outputs.loss.item()


def load_model_for_eval(model_dir: Path, device: str = "cuda") -> HFModelForEval:
    """Load a model for evaluation. Uses HF transformers if config.json is present."""
    config_path = model_dir / "config.json"
    if config_path.exists():
        return HFModelForEval(model_dir, device)
    raise FileNotFoundError(
        f"No config.json found in {model_dir}. "
        "The king model must include a config.json for the eval worker to load it."
    )

# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(cfg: dict) -> None:
    challenge_id = cfg["challenge_id"]
    N = cfg["N"]
    alpha = cfg["alpha"]
    seq_len = cfg["sequence_length"]
    use_amp = cfg.get("use_amp", True)
    amp_dtype = cfg.get("amp_dtype", "bfloat16")

    K = int(binom.isf(alpha, N, 0.5))
    logger.info("Challenge %s: N=%d, K=%d, alpha=%s", challenge_id, N, K, alpha)

    r2 = R2(
        cfg["r2_endpoint_url"], cfg["r2_bucket_name"],
        cfg["r2_access_key_id"], cfg["r2_secret_access_key"],
    )

    eval_prefix = f"eval/{challenge_id}"
    outcomes_key = f"{eval_prefix}/outcomes.jsonl"
    verdict_key = f"{eval_prefix}/verdict.json"
    meta_key = f"{eval_prefix}/meta.json"

    # Write meta
    r2.put_json(meta_key, {
        "challenge_id": challenge_id,
        "king_repo": cfg["king_repo"],
        "challenger_repo": cfg["challenger_repo"],
        "N": N, "K": K, "alpha": alpha,
        "sequence_length": seq_len,
    })

    # Download models
    logger.info("Downloading king from %s", cfg["king_repo"])
    king_dir = download_model(cfg["king_repo"], "/tmp/koth/king", cfg.get("hf_token", ""))

    logger.info("Downloading challenger from %s", cfg["challenger_repo"])
    challenger_dir = download_model(cfg["challenger_repo"], "/tmp/koth/challenger", cfg.get("hf_token", ""))

    # Load state dicts for validation
    logger.info("Loading state dicts for validation...")
    king_sd = load_state_dict(king_dir)
    challenger_sd = load_state_dict(challenger_dir)

    # Validate
    valid, reason, detail = validate_challenger(king_sd, challenger_sd, cfg)
    if not valid:
        logger.error("Validation failed: %s - %s", reason, detail)
        r2.put_json(verdict_key, {
            "accepted": False,
            "verdict": "rejected",
            "reason": reason,
            "detail": detail,
            "challenge_id": challenge_id,
        })
        return

    logger.info("Validation passed, loading models for eval...")
    del king_sd, challenger_sd
    torch.cuda.empty_cache()

    # Load models for forward pass
    king_model = load_model_for_eval(king_dir)
    challenger_model = load_model_for_eval(challenger_dir)

    # Download dataset shard
    shard_key = cfg["dataset_shard_key"]
    shard_path = f"/tmp/koth/eval_shard.npy"
    logger.info("Downloading eval shard: %s", shard_key)
    r2.download_file(shard_key, shard_path)

    # Load dataset
    tokens = np.load(shard_path, mmap_mode="r", allow_pickle=False)
    if tokens.dtype != np.uint32:
        tokens = tokens.astype(np.uint32, copy=False)
    if tokens.ndim != 1:
        tokens = tokens.reshape(-1)
    tokens_t = torch.from_numpy(tokens)

    n_sequences = len(tokens_t) // seq_len
    logger.info("Loaded shard: %d tokens, %d sequences", len(tokens_t), n_sequences)

    # Select eval indices
    seed_material = f"{cfg.get('commit_block_hash', 'default')}:{cfg.get('hotkey', '')}".encode()
    seed_hash = hashlib.blake2b(seed_material, digest_size=8).digest()
    seed = int.from_bytes(seed_hash, "little")
    rng = np.random.Generator(np.random.PCG64(seed))

    actual_N = min(N, n_sequences)
    eval_indices = rng.choice(n_sequences, size=actual_N, replace=False).tolist()

    # Run sign test
    s = 0
    n = 0
    n_ties = 0
    king_loss_sum = 0.0
    challenger_loss_sum = 0.0
    t0 = time.time()
    batch_buffer = []

    logger.info("Starting sign test: N=%d, K=%d", actual_N, K)

    for i, seq_idx in enumerate(eval_indices):
        start = seq_idx * seq_len
        seq_tokens = tokens_t[start : start + seq_len]

        king_loss = king_model.loss_on_tokens(seq_tokens, use_amp=use_amp, amp_dtype=amp_dtype)
        challenger_loss = challenger_model.loss_on_tokens(seq_tokens, use_amp=use_amp, amp_dtype=amp_dtype)

        king_loss_sum += king_loss
        challenger_loss_sum += challenger_loss

        outcome = {
            "seq_idx": seq_idx,
            "king_loss": round(king_loss, 6),
            "challenger_loss": round(challenger_loss, 6),
        }

        if king_loss == challenger_loss:
            outcome["win"] = None
            n_ties += 1
        else:
            n += 1
            win = challenger_loss < king_loss
            if win:
                s += 1
            outcome["win"] = 1 if win else 0

        outcome["s"] = s
        outcome["n"] = n
        outcome["N"] = actual_N
        batch_buffer.append(outcome)

        # Flush to R2 every 100 outcomes
        if len(batch_buffer) >= 100:
            for rec in batch_buffer:
                r2.append_jsonl(outcomes_key, rec)
            batch_buffer = []

        # Early stopping
        if n > 0:
            if s >= K:
                logger.info("EARLY STOP: challenger wins at n=%d, s=%d >= K=%d", n, s, K)
                break
            remaining = actual_N - (n + n_ties)
            if s + remaining < K:
                logger.info("EARLY STOP: king wins at n=%d, s=%d, remaining=%d < needed=%d", n, s, remaining, K - s)
                break

        if (i + 1) % 500 == 0:
            logger.info(
                "Progress: %d/%d evaluated, s=%d, n=%d, win_rate=%.4f",
                i + 1, actual_N, s, n, s / n if n > 0 else 0,
            )

    # Flush remaining
    for rec in batch_buffer:
        r2.append_jsonl(outcomes_key, rec)

    elapsed = time.time() - t0
    total_evaluated = n + n_ties
    accepted = s >= K
    win_rate = s / n if n > 0 else 0.0
    early_stopped = total_evaluated < actual_N

    verdict = {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "S_N": s,
        "K": K,
        "N": actual_N,
        "n_evaluated": n,
        "n_ties": n_ties,
        "win_rate": round(win_rate, 6),
        "alpha": alpha,
        "early_stopped": early_stopped,
        "early_stop_reason": (
            "challenger_reached_K" if accepted and early_stopped
            else "king_unreachable" if not accepted and early_stopped
            else "full_eval"
        ),
        "avg_king_loss": round(king_loss_sum / total_evaluated, 6) if total_evaluated > 0 else 0,
        "avg_challenger_loss": round(challenger_loss_sum / total_evaluated, 6) if total_evaluated > 0 else 0,
        "wall_time_s": round(elapsed, 1),
        "challenge_id": challenge_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    r2.put_json(verdict_key, verdict)
    logger.info("Verdict written: %s (s=%d, K=%d, win_rate=%.4f, time=%.1fs)", verdict["verdict"], s, K, win_rate, elapsed)


def main():
    parser = argparse.ArgumentParser(description="KOTH eval worker")
    parser.add_argument("--config", required=True, help="Path to eval config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    run_eval(cfg)


if __name__ == "__main__":
    main()
