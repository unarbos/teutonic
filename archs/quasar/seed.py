#!/usr/bin/env python3
"""Seed a freshly initialised Quasar hybrid MoE checkpoint.

Architecture: SILX AI Quasar (silx-ai/Quasar-3B-A1B-Preview) — looped hybrid
transformer with Quasar+GLA attention layers, persistent latent memory, and
SMEBU BigMac MoE. The vendored modeling code lives in `archs/quasar/` so we
do NOT use trust_remote_code anywhere; the seed config strips auto_map before
push so consumers load via plain AutoModelForCausalLM after importing the
arch package (handled by `chain_config.load_arch()`).

Sizing (verified by `archs/quasar/size.py`):
    d_model=4096, n_layers=32, n_heads=32, head_dim=128
    quasar_layers=4, gated_layers=2 (cycle), dense_input_layers=4
    num_routed_experts=56, top_k=8, routed_expert_size=1024 (effective; BigMac
    DCCA inflates to 4096 internally), shared_expert_size=2048, bigmac_r=0.25
    memory_slots=128, memory_dim=128, num_loops=1
    => 7.962B active / 24.873B total

Default target repo, upload backend, and tokenizer come from chain.toml
([chain].seed_repo plus [seed].repo_backend / [seed].tokenizer_repo). Override
via TEUTONIC_SEED_REPO_OVERRIDE / TEUTONIC_SEED_REPO_BACKEND_OVERRIDE /
TEUTONIC_SEED_TOKENIZER_OVERRIDE if you want to push elsewhere.

Run on the GPU box (CPU init of 24B params takes ~12 min; on a B200 it is
seconds via vectorized RNG):
    source /home/const/workspace/.venv/bin/activate
    python scripts/seed.py [--push] [--no-probe]
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
from model_store import upload_model_folder

chain_config.load_arch()
from archs.quasar import QuasarConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seed")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
TARGET_REPO = os.environ.get("TEUTONIC_SEED_REPO_OVERRIDE", chain_config.SEED_REPO)
TARGET_BACKEND = os.environ.get(
    "TEUTONIC_SEED_REPO_BACKEND_OVERRIDE",
    getattr(chain_config, "SEED_REPO_BACKEND", "hf"),
)
_default_tokenizer_repo = chain_config.SEED_TOKENIZER_REPO
if not _default_tokenizer_repo and TARGET_BACKEND == "hf":
    _default_tokenizer_repo = chain_config.SEED_REPO
TOKENIZER_REPO = os.environ.get("TEUTONIC_SEED_TOKENIZER_OVERRIDE",
                                _default_tokenizer_repo)
OUT_DIR = os.environ.get("TEUTONIC_SEED_DIR",
                         f"/tmp/{chain_config.NAME.lower()}")

VOCAB_SIZE = int(os.environ.get("TEUTONIC_SEED_VOCAB", "262144"))
HIDDEN_SIZE = int(os.environ.get("TEUTONIC_SEED_HIDDEN", "4096"))
NUM_LAYERS = int(os.environ.get("TEUTONIC_SEED_NUM_LAYERS", "32"))
NUM_HEADS = int(os.environ.get("TEUTONIC_SEED_HEADS", "32"))
HEAD_DIM = int(os.environ.get("TEUTONIC_SEED_HEAD_DIM", "128"))
D_FF = int(os.environ.get("TEUTONIC_SEED_D_FF", "11008"))
QUASAR_LAYERS = int(os.environ.get("TEUTONIC_SEED_QUASAR_LAYERS", "4"))
GATED_LAYERS = int(os.environ.get("TEUTONIC_SEED_GATED_LAYERS", "2"))
DENSE_INPUT_LAYERS = int(os.environ.get("TEUTONIC_SEED_DENSE_INPUT_LAYERS", "4"))
NUM_ROUTED_EXPERTS = int(os.environ.get("TEUTONIC_SEED_NUM_ROUTED_EXPERTS", "56"))
NUM_SHARED_EXPERTS = int(os.environ.get("TEUTONIC_SEED_NUM_SHARED_EXPERTS", "1"))
TOP_K = int(os.environ.get("TEUTONIC_SEED_TOP_K", "8"))
ROUTED_EXPERT_SIZE = int(os.environ.get("TEUTONIC_SEED_ROUTED_EXPERT_SIZE", "1024"))
SHARED_EXPERT_SIZE = int(os.environ.get("TEUTONIC_SEED_SHARED_EXPERT_SIZE", "2048"))
BIGMAC_R = float(os.environ.get("TEUTONIC_SEED_BIGMAC_R", "0.25"))
MEMORY_SLOTS = int(os.environ.get("TEUTONIC_SEED_MEMORY_SLOTS", "128"))
MEMORY_DIM = int(os.environ.get("TEUTONIC_SEED_MEMORY_DIM", "128"))
NUM_LOOPS = int(os.environ.get("TEUTONIC_SEED_NUM_LOOPS", "1"))
MAX_SEQ_LEN = int(os.environ.get("TEUTONIC_SEED_MAX_SEQ_LEN", "16384"))
ROPE_THETA = float(os.environ.get("TEUTONIC_SEED_ROPE_THETA", "1000000.0"))


def build_config() -> QuasarConfig:
    """Quasar 8B-active / ~24B-total config; tokenizer pinned to chain.toml seed.tokenizer_repo."""
    cfg = QuasarConfig(
        vocab_size=VOCAB_SIZE,
        d_model=HIDDEN_SIZE,
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        d_ff=D_FF,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        tie_word_embeddings=True,
        quasar_layers=QUASAR_LAYERS,
        gated_layers=GATED_LAYERS,
        memory_slots=MEMORY_SLOTS,
        memory_dim=MEMORY_DIM,
        moe_type="bigmac",
        num_shared_experts=NUM_SHARED_EXPERTS,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        top_k=TOP_K,
        shared_expert_size=SHARED_EXPERT_SIZE,
        routed_expert_size=ROUTED_EXPERT_SIZE,
        dense_input_layers=DENSE_INPUT_LAYERS,
        bigmac_r=BIGMAC_R,
        num_loops=NUM_LOOPS,
        rope_theta=ROPE_THETA,
        # Default tokenizer is Gemma3-derived: pad=0, eos=1, bos=2.
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        num_key_value_heads=NUM_HEADS,
    )
    cfg.architectures = ["QuasarForCausalLM"]
    return cfg


def _strip_auto_map(out_dir: Path):
    """Remove auto_map from config.json so consumers never invoke remote code.

    The vendored archs/quasar package handles loading via the standard
    AutoModelForCausalLM dispatch once imported; auto_map would cause HF to
    attempt a dynamic import of the original silx-ai modules.
    """
    cfg_path = out_dir / "config.json"
    with open(cfg_path) as f:
        cfg_data = json.load(f)
    cfg_data.pop("auto_map", None)
    with open(cfg_path, "w") as f:
        json.dump(cfg_data, f, indent=2)


def _count_active(model, cfg: QuasarConfig) -> tuple[int, int]:
    """Return (total, active_per_token) parameter counts. Mirrors archs/quasar/size.py."""
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
        if "experts_w12" in name or "experts_w3" in name:
            num_experts = p.shape[0]
            active += (n // num_experts) * cfg.top_k
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
    parser.add_argument("--repo-backend", choices=("hf", "hippius"),
                        default=TARGET_BACKEND,
                        help=f"upload backend for {TARGET_REPO} (default: {TARGET_BACKEND})")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip on-GPU trainability probe")
    parser.add_argument("--device", default="cuda:0",
                        help="GPU for the probe (default cuda:0)")
    args = parser.parse_args()

    if not TOKENIZER_REPO:
        raise SystemExit(
            "No tokenizer repo configured. Set [seed].tokenizer_repo in chain.toml "
            "or TEUTONIC_SEED_TOKENIZER_OVERRIDE when [seed].repo_backend = \"hippius\"."
        )

    out = Path(OUT_DIR)
    if out.exists():
        log.info("clearing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cfg = build_config()
    log.info("config: d_model=%d n_layers=%d n_heads=%d head_dim=%d d_ff=%d vocab=%d",
             cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.head_dim,
             cfg.d_ff, cfg.vocab_size)
    log.info("hybrid: quasar_layers=%d gated_layers=%d dense_input_layers=%d num_loops=%d",
             cfg.quasar_layers, cfg.gated_layers, cfg.dense_input_layers,
             cfg.num_loops)
    log.info("moe: num_routed_experts=%d top_k=%d routed_expert_size=%d shared_expert_size=%d bigmac_r=%.2f",
             cfg.num_routed_experts, cfg.top_k, cfg.routed_expert_size,
             cfg.shared_expert_size, cfg.bigmac_r)
    log.info("memory: slots=%d dim=%d  rope_theta=%.0f  tie_embed=%s",
             cfg.memory_slots, cfg.memory_dim, cfg.rope_theta,
             cfg.tie_word_embeddings)

    log.info("instantiating model from config (random init, bf16)")
    t0 = time.time()
    torch.manual_seed(0xC0DE)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    total, active = _count_active(model, cfg)
    log.info("model built: total=%.3fB / active=%.3fB params in %.1fs",
             total / 1e9, active / 1e9, time.time() - t0)

    # The Quasar config exposes `top_k` for MoE expert selection. HF copies
    # this onto generation_config, where it gets validated as a sampling
    # top_k that requires do_sample=True. Strip the bogus generation-side
    # top_k so save_pretrained does not refuse to write generation_config.
    if model.generation_config is not None:
        model.generation_config.top_k = None

    log.info("saving model to %s", out)
    model.save_pretrained(out, safe_serialization=True)
    del model
    _strip_auto_map(out)
    log.info("stripped auto_map from config.json (no trust_remote_code path)")

    log.info("downloading tokenizer from %s", TOKENIZER_REPO)
    tok_dir = Path(f"{OUT_DIR}-tokenizer")
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
        from eval.torch_runner import trainability_probe
        m = AutoModelForCausalLM.from_pretrained(
            out, dtype=torch.bfloat16, device_map={"": args.device},
            attn_implementation="eager",
        )
        m.eval()
        if hasattr(m, "reset_state"):
            m.reset_state()
        probe = trainability_probe(m)
        log.info("probe: ok=%s before=%.4f after=%.4f delta=%.4f reason=%s",
                 probe["ok"], probe["loss_before"], probe["loss_after"],
                 probe["delta"], probe["reason"])
        del m
        torch.cuda.empty_cache()
        if not probe["ok"]:
            log.error("probe FAILED — aborting before push")
            sys.exit(1)

    if args.push and args.repo_backend == "hippius":
        if args.public:
            log.warning("--public has no effect for Hippius Hub uploads")
        log.info("uploading folder %s -> Hippius %s", out, TARGET_REPO)
        ref = upload_model_folder(
            out,
            repo=TARGET_REPO,
            commit_message=(
                f"seed {chain_config.NAME} (Quasar n_layers={NUM_LAYERS} d_model={HIDDEN_SIZE} "
                f"experts={NUM_ROUTED_EXPERTS}/top{TOP_K}, fresh init)"
            ),
        )
        log.info("uploaded to %s", ref.immutable_ref)
        print(f"SEED_REPO={ref.repo}")
        print(f"SEED_DIGEST={ref.digest}")
    elif args.push:
        api = HfApi(token=HF_TOKEN or None)
        private = not args.public
        log.info("creating/updating repo %s (private=%s)", TARGET_REPO, private)
        api.create_repo(TARGET_REPO, exist_ok=True, private=private,
                        repo_type="model")
        log.info("uploading folder %s -> %s", out, TARGET_REPO)
        api.upload_folder(
            folder_path=str(out),
            repo_id=TARGET_REPO,
            commit_message=(
                f"seed {chain_config.NAME} (Quasar n_layers={NUM_LAYERS} d_model={HIDDEN_SIZE} "
                f"experts={NUM_ROUTED_EXPERTS}/top{TOP_K}, fresh init)"
            ),
            allow_patterns=[
                "*.safetensors", "*.json", "tokenizer*", "special_tokens*",
            ],
            # No *.py upload — vendored modeling code lives only in archs/quasar/.
        )
        info = api.model_info(TARGET_REPO)
        log.info("uploaded to %s @ hf:%s", TARGET_REPO, info.sha)
        print(f"SEED_REPO={TARGET_REPO}")
        print(f"SEED_DIGEST=hf:{info.sha}")
    else:
        log.info("skipped push (use --push to upload)")

    cfg_path = out / "config.json"
    with open(cfg_path) as f:
        log.info("final config.json:\n%s", f.read())

    log.info("done. local dir: %s  total=%.3fB active=%.3fB",
             out, total / 1e9, active / 1e9)


if __name__ == "__main__":
    main()
