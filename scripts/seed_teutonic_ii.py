#!/usr/bin/env python3
"""Seed unconst/Teutonic-II — a freshly initialized Gemma3 ~4B model.

Architecture mirrors HF transformers' Gemma3TextConfig defaults (which match
google/gemma-3-4b-pt) so the model is recognized as a standard Gemma3 4B by
the eval pipeline. Tokenizer is reused from unconst/Teutonic-I (vocab=262144)
so we keep the existing dataset shards 1:1 valid (tokens are integer IDs in
the same vocab space).

The model is created with random init via AutoModelForCausalLM.from_config
(no pretrained weights) — fairness requires that the seed king be beatable.

Run:
    source /home/const/workspace/.venv/bin/activate
    python teutonic/scripts/seed_teutonic_ii.py [--push] [--no-probe]
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3TextConfig,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seed_teutonic_ii")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
TARGET_REPO = os.environ.get("TEUTONIC_II_REPO", "unconst/Teutonic-II")
TOKENIZER_REPO = os.environ.get("TEUTONIC_II_TOKENIZER", "unconst/Teutonic-I")
OUT_DIR = os.environ.get("TEUTONIC_II_DIR", "/tmp/teutonic-ii")


def build_config() -> Gemma3TextConfig:
    """Gemma3 4B-shaped config, vocab pinned to Teutonic-I (262144)."""
    cfg = Gemma3TextConfig(
        vocab_size=262144,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=106,
        bos_token_id=2,
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=1024,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        use_bidirectional_attention=False,
        dtype="bfloat16",
    )
    cfg.architectures = ["Gemma3ForCausalLM"]
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true",
                        help="upload to TARGET_REPO as private")
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

    log.info("instantiating model from config (random init, bf16)")
    t0 = time.time()
    torch.manual_seed(0xC0DE)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model built: %.2fB params in %.1fs",
             n_params / 1e9, time.time() - t0)

    log.info("saving model to %s", out)
    model.save_pretrained(out, safe_serialization=True)

    log.info("downloading tokenizer from %s", TOKENIZER_REPO)
    tok_dir = Path("/tmp/teutonic-ii-tokenizer")
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
        log.info("creating/updating private repo %s", TARGET_REPO)
        api.create_repo(TARGET_REPO, exist_ok=True, private=True,
                        repo_type="model")
        log.info("uploading folder %s -> %s", out, TARGET_REPO)
        api.upload_folder(
            folder_path=str(out),
            repo_id=TARGET_REPO,
            commit_message="seed Teutonic-II (Gemma3 4B, fresh init)",
            allow_patterns=[
                "*.safetensors", "*.json", "tokenizer*", "special_tokens*",
            ],
        )
        log.info("uploaded. NOT made public yet — flip at cutover.")
    else:
        log.info("skipped push (use --push to upload)")

    cfg_path = out / "config.json"
    with open(cfg_path) as f:
        log.info("final config.json:\n%s", f.read())

    log.info("done. local dir: %s", out)


if __name__ == "__main__":
    main()
