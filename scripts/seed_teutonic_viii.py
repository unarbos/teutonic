#!/usr/bin/env python3
"""Seed unconst/Teutonic-VIII — a freshly initialized Qwen3 ~8B model.

Architecture: HF transformers' Qwen3Config with hidden=4096, 36 layers,
ffn=12288, 32/8 heads — lands at 8.02B params with vocab pinned to 262144
(same tokenizer as Teutonic-I/III). All other defaults (per-head q/k norm,
RoPE theta=1e6, RMSNorm) are kept.

Tokenizer is reused from unconst/Teutonic-I (vocab=262144) so the existing
dataset shards stay 1:1 valid (token IDs are integers in the same vocab
space).

The model is created with random init via AutoModelForCausalLM.from_config
(no pretrained weights) — fairness requires that the seed king be beatable.

Run:
    source /home/const/workspace/.venv/bin/activate
    python teutonic/scripts/seed_teutonic_viii.py [--push] [--no-probe]
"""
import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seed_teutonic_viii")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
TARGET_REPO = os.environ.get("TEUTONIC_VIII_REPO", "unconst/Teutonic-VIII")
TOKENIZER_REPO = os.environ.get("TEUTONIC_VIII_TOKENIZER", "unconst/Teutonic-I")
OUT_DIR = os.environ.get("TEUTONIC_VIII_DIR", "/tmp/teutonic-viii")

NUM_LAYERS = int(os.environ.get("TEUTONIC_VIII_NUM_LAYERS", "36"))
HIDDEN_SIZE = int(os.environ.get("TEUTONIC_VIII_HIDDEN", "4096"))
INTERMEDIATE = int(os.environ.get("TEUTONIC_VIII_INTERMEDIATE", "12288"))
NUM_HEADS = int(os.environ.get("TEUTONIC_VIII_HEADS", "32"))
NUM_KV_HEADS = int(os.environ.get("TEUTONIC_VIII_KV_HEADS", "8"))
HEAD_DIM = int(os.environ.get("TEUTONIC_VIII_HEAD_DIM", "128"))


def build_config() -> Qwen3Config:
    """Qwen3 8B-shaped config, vocab pinned to Teutonic-I (262144)."""
    cfg = Qwen3Config(
        vocab_size=262144,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=131072,
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        dtype="bfloat16",
    )
    cfg.architectures = ["Qwen3ForCausalLM"]
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true",
                        help="upload to TARGET_REPO (private by default)")
    parser.add_argument("--public", action="store_true",
                        help="when --push, create public (overrides default private)")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip on-GPU trainability probe")
    parser.add_argument("--device", default="cuda:0",
                        help="GPU for the probe (default cuda:0)")
    args = parser.parse_args()

    out = Path(OUT_DIR)
    if out.exists():
        log.info("clearing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cfg = build_config()
    log.info("config: hidden=%d layers=%d heads=%d kv=%d ffn=%d vocab=%d",
             cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
             cfg.num_key_value_heads, cfg.intermediate_size, cfg.vocab_size)
    log.info("rope_theta=%.0f tie_embed=%s",
             getattr(cfg, "rope_theta", 0.0), cfg.tie_word_embeddings)

    log.info("instantiating model from config (random init, bf16)")
    t0 = time.time()
    torch.manual_seed(0xC0DE)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model built: %.3fB params in %.1fs",
             n_params / 1e9, time.time() - t0)

    log.info("saving model to %s", out)
    model.save_pretrained(out, safe_serialization=True)
    del model

    log.info("downloading tokenizer from %s", TOKENIZER_REPO)
    tok_dir = Path("/tmp/teutonic-viii-tokenizer")
    if tok_dir.exists():
        shutil.rmtree(tok_dir)
    snapshot_download(
        TOKENIZER_REPO, local_dir=str(tok_dir),
        token=HF_TOKEN or None,
        allow_patterns=["tokenizer*", "special_tokens*"],
    )
    tok = AutoTokenizer.from_pretrained(str(tok_dir),
                                        token=HF_TOKEN or None)
    if tok.vocab_size != cfg.vocab_size and len(tok) != cfg.vocab_size:
        log.warning("tokenizer vocab=%d/%d differs from config vocab=%d",
                    tok.vocab_size, len(tok), cfg.vocab_size)
    tok.save_pretrained(out)
    log.info("tokenizer saved (vocab=%d)", len(tok))

    if not args.no_probe:
        log.info("running trainability probe on %s", args.device)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from eval_torch import trainability_probe
        m = AutoModelForCausalLM.from_pretrained(
            out, dtype=torch.bfloat16, device_map={"": args.device},
            attn_implementation="sdpa",
        )
        m.eval()
        probe = trainability_probe(m)
        log.info("probe: ok=%s before=%.4f after=%.4f delta=%.4f reason=%s",
                 probe["ok"], probe["loss_before"], probe["loss_after"],
                 probe["delta"], probe["reason"])
        del m
        torch.cuda.empty_cache()
        if not probe["ok"]:
            log.error("probe FAILED — aborting before push")
            sys.exit(1)

    if args.push:
        api = HfApi(token=HF_TOKEN or None)
        private = not args.public
        log.info("creating/updating repo %s (private=%s)", TARGET_REPO, private)
        api.create_repo(TARGET_REPO, exist_ok=True, private=private,
                        repo_type="model")
        log.info("uploading folder %s -> %s", out, TARGET_REPO)
        api.upload_folder(
            folder_path=str(out),
            repo_id=TARGET_REPO,
            commit_message=f"seed Teutonic-VIII (Qwen3 {NUM_LAYERS}L hidden={HIDDEN_SIZE}, fresh init)",
            allow_patterns=[
                "*.safetensors", "*.json", "tokenizer*", "special_tokens*",
            ],
        )
        log.info("uploaded.")
    else:
        log.info("skipped push (use --push to upload)")

    cfg_path = out / "config.json"
    with open(cfg_path) as f:
        log.info("final config.json:\n%s", f.read())

    log.info("done. local dir: %s", out)


if __name__ == "__main__":
    main()
