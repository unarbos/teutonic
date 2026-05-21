#!/usr/bin/env python3
"""Seed a freshly initialised Qwen3-MoE checkpoint for Teutonic-LXXX.

Architecture: vanilla `transformers.Qwen3MoeForCausalLM` (no custom modeling).
Sizing default (verified by `archs/qwen3_moe/size.py`):
    vocab=262144 (Teutonic-I / Gemma3-derived — same as Quasar)
    hidden=4096, layers=36, heads=32, kv_heads=8, head_dim=128
    intermediate_size=11008, num_experts=128, top_k=8, moe_intermediate_size=1408
    => 82.33B total / 7.59B active

Default target repo + tokenizer come from chain.toml ([chain].seed_repo and
[seed].tokenizer_repo). On the live LXXX chain that's
`unconst/Teutonic-LXXX-mock-king` + `unconst/Teutonic-I`. Override the push
target via TEUTONIC_SEED_REPO_OVERRIDE if you want to seed a sibling repo
(e.g. for a fresh genesis bake without overwriting the live king).

Run on the GPU box (random init is fast — under a minute on one B300):
    source /workspace/teutonic/.venv/bin/activate
    TEUTONIC_SEED_REPO_OVERRIDE=unconst/Teutonic-LXXX-fresh \
    HF_HOME=/workspace/hf-cache \
    python scripts/seed.py --push --no-probe
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import chain_config

chain_config.load_arch()
from archs.qwen3_moe import Qwen3MoeConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seed-qwen3moe")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
TARGET_REPO = os.environ.get("TEUTONIC_SEED_REPO_OVERRIDE", chain_config.SEED_REPO)
TOKENIZER_REPO = os.environ.get("TEUTONIC_SEED_TOKENIZER_OVERRIDE",
                                chain_config.SEED_TOKENIZER_REPO or chain_config.SEED_REPO)
OUT_DIR = os.environ.get("TEUTONIC_SEED_DIR",
                         f"/tmp/{chain_config.NAME.lower()}")

_default_vocab = "151936" if "Qwen" in (TOKENIZER_REPO or "") else "262144"
VOCAB_SIZE = int(os.environ.get("TEUTONIC_SEED_VOCAB", _default_vocab))
HIDDEN_SIZE = int(os.environ.get("TEUTONIC_SEED_HIDDEN", "4096"))
NUM_LAYERS = int(os.environ.get("TEUTONIC_SEED_NUM_LAYERS", "36"))
NUM_HEADS = int(os.environ.get("TEUTONIC_SEED_HEADS", "32"))
NUM_KV_HEADS = int(os.environ.get("TEUTONIC_SEED_KV_HEADS", "8"))
HEAD_DIM = int(os.environ.get("TEUTONIC_SEED_HEAD_DIM", "128"))
INTERMEDIATE_SIZE = int(os.environ.get("TEUTONIC_SEED_INTERMEDIATE", "11008"))
NUM_EXPERTS = int(os.environ.get("TEUTONIC_SEED_NUM_EXPERTS", "128"))
TOP_K = int(os.environ.get("TEUTONIC_SEED_TOP_K", "8"))
MOE_INTERMEDIATE_SIZE = int(os.environ.get("TEUTONIC_SEED_MOE_INTERMEDIATE", "1408"))
DECODER_SPARSE_STEP = int(os.environ.get("TEUTONIC_SEED_DECODER_SPARSE_STEP", "1"))
ROUTER_AUX_LOSS_COEF = float(os.environ.get("TEUTONIC_SEED_ROUTER_AUX_LOSS", "0.001"))
MAX_SEQ_LEN = int(os.environ.get("TEUTONIC_SEED_MAX_SEQ_LEN", "16384"))
ROPE_THETA = float(os.environ.get("TEUTONIC_SEED_ROPE_THETA", "1000000.0"))


def build_config() -> Qwen3MoeConfig:
    cfg = Qwen3MoeConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=MAX_SEQ_LEN,
        tie_word_embeddings=True,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
        decoder_sparse_step=DECODER_SPARSE_STEP,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=ROUTER_AUX_LOSS_COEF,
        mlp_only_layers=[],
        rope_theta=ROPE_THETA,
        # Tokenizer special tokens. Default to Gemma3 (Teutonic-I): pad=0,
        # eos=1, bos=2 — matches Quasar / live `dataset/v2/` shards. Override
        # via TEUTONIC_SEED_{BOS,EOS,PAD}_ID to seed under a different
        # tokenizer (e.g. Qwen: bos=151643, eos=151645, pad=151643).
        bos_token_id=int(os.environ.get(
            "TEUTONIC_SEED_BOS_ID",
            "151643" if VOCAB_SIZE == 151936 else "2",
        )),
        eos_token_id=int(os.environ.get(
            "TEUTONIC_SEED_EOS_ID",
            "151645" if VOCAB_SIZE == 151936 else "1",
        )),
        pad_token_id=int(os.environ.get(
            "TEUTONIC_SEED_PAD_ID",
            "151643" if VOCAB_SIZE == 151936 else "0",
        )),
    )
    cfg.architectures = ["Qwen3MoeForCausalLM"]
    return cfg


def _strip_auto_map(out_dir: Path):
    """Remove auto_map from config.json so consumers never invoke remote code.
    For vanilla Qwen3MoE this is a no-op (HF Auto* dispatches via model_type),
    but we keep parity with archs/quasar/seed.py for defense in depth."""
    cfg_path = out_dir / "config.json"
    with open(cfg_path) as f:
        cfg_data = json.load(f)
    if cfg_data.pop("auto_map", None) is not None:
        with open(cfg_path, "w") as f:
            json.dump(cfg_data, f, indent=2)
        log.info("stripped auto_map from %s", cfg_path)


def _count_active(model, cfg: Qwen3MoeConfig) -> tuple[int, int]:
    """Mirror archs/qwen3_moe/size.count_params, on a real model."""
    total = 0
    active = 0
    embed_seen = False
    tied = bool(cfg.tie_word_embeddings)
    for name, p in model.named_parameters():
        n = p.numel()
        if "lm_head" in name and tied:
            continue
        if "embed_tokens" in name and tied and embed_seen:
            continue
        if "embed_tokens" in name:
            embed_seen = True
        total += n
        if ".experts." in name:
            active += int(n * cfg.num_experts_per_tok / cfg.num_experts)
        else:
            active += n
    return total, active


def main():
    parser = argparse.ArgumentParser(
        description=f"Seed the genesis king for chain {chain_config.NAME}")
    parser.add_argument("--push", action="store_true",
                        help=f"upload to {TARGET_REPO} (private by default)")
    parser.add_argument("--public", action="store_true",
                        help="when --push, create public (overrides default private)")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip on-GPU trainability probe (probe assumes single-GPU; "
                             "for sharded 80B you must currently --no-probe)")
    parser.add_argument("--device", default="cuda:0",
                        help="GPU for the (optional) probe — single GPU only for now")
    parser.add_argument("--hippius", action="store_true",
                        help="upload to Hippius Hub (registry.hippius.com) instead of HF")
    args = parser.parse_args()

    out = Path(OUT_DIR)
    if out.exists():
        log.info("clearing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cfg = build_config()
    log.info("config: hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d intermediate=%d vocab=%d",
             cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
             cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size,
             cfg.vocab_size)
    log.info("moe: num_experts=%d top_k=%d moe_intermediate=%d sparse_step=%d",
             cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size,
             cfg.decoder_sparse_step)
    # transformers 5.5+ flattens rope into a `rope_parameters` dict; older
    # versions exposed `rope_theta` as a top-level attribute.
    _rope = getattr(cfg, "rope_parameters", None) or {"rope_theta": getattr(cfg, "rope_theta", None)}
    log.info("rope=%s tie_embed=%s max_pos=%d",
             _rope, cfg.tie_word_embeddings, cfg.max_position_embeddings)

    log.info("instantiating model from config (random init, bf16) ...")
    t0 = time.time()
    torch.manual_seed(0xC0DE)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    total, active = _count_active(model, cfg)
    log.info("model built: total=%.3fB / active=%.3fB params in %.1fs",
             total / 1e9, active / 1e9, time.time() - t0)

    if model.generation_config is not None:
        # Same defensive scrub as Quasar — strip MoE top_k from generation_config.
        if getattr(model.generation_config, "top_k", None) == cfg.num_experts_per_tok:
            model.generation_config.top_k = None

    log.info("saving model to %s", out)
    t1 = time.time()
    model.save_pretrained(out, safe_serialization=True)
    log.info("saved in %.1fs", time.time() - t1)
    del model
    _strip_auto_map(out)

    log.info("downloading tokenizer from %s", TOKENIZER_REPO)
    tok_dir = Path(f"{OUT_DIR}-tokenizer")
    if tok_dir.exists():
        shutil.rmtree(tok_dir)
    snapshot_download(
        TOKENIZER_REPO, local_dir=str(tok_dir),
        token=HF_TOKEN or None,
        allow_patterns=["tokenizer*", "special_tokens*", "vocab*", "merges*"],
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
        from eval.torch_runner import trainability_probe
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
    else:
        log.info("skipping trainability probe (--no-probe)")

    if args.push and args.hippius:
        from model_store import upload_model_folder

        log.info("uploading folder %s -> Hippius %s", out, TARGET_REPO)
        t2 = time.time()
        ref = upload_model_folder(
            out,
            repo=TARGET_REPO,
            commit_message=(
                f"seed {chain_config.NAME} (Qwen3MoE layers={NUM_LAYERS} "
                f"hidden={HIDDEN_SIZE} experts={NUM_EXPERTS}/top{TOP_K}, fresh init)"
            ),
        )
        log.info("hippius upload done in %.1fs digest=%s", time.time() - t2, ref.digest)
        print(f"SEED_DIGEST={ref.digest}")
    elif args.push:
        api = HfApi(token=HF_TOKEN or None)
        private = not args.public
        log.info("creating/updating repo %s (private=%s)", TARGET_REPO, private)
        api.create_repo(TARGET_REPO, exist_ok=True, private=private,
                        repo_type="model")
        log.info("uploading folder %s -> %s", out, TARGET_REPO)
        t2 = time.time()
        api.upload_folder(
            folder_path=str(out),
            repo_id=TARGET_REPO,
            commit_message=(
                f"seed {chain_config.NAME} (Qwen3MoE layers={NUM_LAYERS} "
                f"hidden={HIDDEN_SIZE} experts={NUM_EXPERTS}/top{TOP_K}, fresh init)"
            ),
            allow_patterns=[
                "*.safetensors", "*.json", "tokenizer*", "special_tokens*",
                "vocab*", "merges*",
            ],
        )
        log.info("uploaded in %.1fs", time.time() - t2)
    else:
        log.info("skipped push (use --push [--hippius] to upload)")

    cfg_path = out / "config.json"
    with open(cfg_path) as f:
        log.info("final config.json:\n%s", f.read())

    log.info("done. local dir: %s  total=%.3fB active=%.3fB",
             out, total / 1e9, active / 1e9)


if __name__ == "__main__":
    main()
