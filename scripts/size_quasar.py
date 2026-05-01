#!/usr/bin/env python3
"""Quasar config sizer. Builds QuasarConfig + QuasarForCausalLM on the meta
device and reports total / active parameter counts. Iterate with --hidden /
--n-layers / --num-experts / --top-k / --routed-expert-size / --shared-expert-size
until the printed numbers land near 8B active / 24B total.

Usage:
    source /home/const/workspace/.venv/bin/activate
    python teutonic/scripts/size_quasar.py \
        --hidden 4096 --n-layers 32 --num-experts 80 --top-k 10 \
        --routed-expert-size 2560 --shared-expert-size 4096 --d-ff 11008
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from accelerate import init_empty_weights

import teutonic.quasar  # registers QuasarConfig with AutoConfig
from teutonic.quasar import QuasarConfig, QuasarForCausalLM


def build_config(args) -> QuasarConfig:
    return QuasarConfig(
        vocab_size=args.vocab_size,
        d_model=args.hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        tie_word_embeddings=args.tie_word_embeddings,
        quasar_layers=args.quasar_layers,
        gated_layers=args.gated_layers,
        memory_slots=args.memory_slots,
        memory_dim=args.memory_dim,
        moe_type="bigmac",
        num_shared_experts=args.num_shared_experts,
        num_routed_experts=args.num_experts,
        top_k=args.top_k,
        shared_expert_size=args.shared_expert_size,
        routed_expert_size=args.routed_expert_size,
        dense_input_layers=args.dense_input_layers,
        bigmac_r=args.bigmac_r,
        rope_theta=args.rope_theta,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
        num_key_value_heads=args.n_heads,
    )


def count_params(model, cfg: QuasarConfig):
    """Walk the meta model and split each parameter into total/active."""
    total = 0
    active = 0
    by_class: dict[str, int] = {}
    by_class_active: dict[str, int] = {}

    embed_counted_once = False
    tied = bool(cfg.tie_word_embeddings)

    for name, p in model.named_parameters():
        n = p.numel()

        if "lm_head" in name and tied:
            continue
        if "embed_tokens" in name and tied and embed_counted_once:
            continue
        if "embed_tokens" in name:
            embed_counted_once = True

        bucket = _classify(name)
        by_class[bucket] = by_class.get(bucket, 0) + n
        total += n

        if "experts_w12" in name or "experts_w3" in name:
            shape = p.shape
            num_experts = shape[0]
            per_expert = n // num_experts
            active_n = per_expert * cfg.top_k
            active += active_n
            by_class_active[bucket] = by_class_active.get(bucket, 0) + active_n
        else:
            active += n
            by_class_active[bucket] = by_class_active.get(bucket, 0) + n

    return total, active, by_class, by_class_active


def _classify(name: str) -> str:
    n = name.lower()
    if "embed_tokens" in n:
        return "embed"
    if "lm_head" in n:
        return "lm_head"
    if "experts_w12" in n or "experts_w3" in n:
        return "moe_experts_routed"
    if "shared_experts" in n:
        return "moe_experts_shared"
    if "w_down_proj" in n or "w_up_proj" in n:
        return "moe_dcca"
    if "router" in n:
        return "moe_router"
    if "moe_bias" in n or "moe_momentum" in n or "max_vio" in n:
        return "moe_smebu_buffers"
    if ".memory." in n or ".W_alpha" in n or ".C_to_hidden" in n:
        return "latent_memory"
    if "ffn.gate" in n or "ffn.up" in n or "ffn.down" in n:
        return "ffn_dense"
    if "attn." in n or "_proj" in n:
        return "attn"
    if "norm" in n or "ln1" in n or "ln2" in n:
        return "norm"
    if "rotary_emb" in n:
        return "rope"
    if "injection_gate" in n:
        return "injection"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=262144)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--n-layers", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=11008)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=16384)
    p.add_argument("--tie-word-embeddings", action="store_true", default=True)
    p.add_argument("--no-tie", dest="tie_word_embeddings", action="store_false")
    p.add_argument("--quasar-layers", type=int, default=4)
    p.add_argument("--gated-layers", type=int, default=2)
    p.add_argument("--memory-slots", type=int, default=128)
    p.add_argument("--memory-dim", type=int, default=128)
    p.add_argument("--num-shared-experts", type=int, default=1)
    p.add_argument("--num-experts", type=int, default=80,
                   help="num_routed_experts")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--shared-expert-size", type=int, default=4096)
    p.add_argument("--routed-expert-size", type=int, default=2560)
    p.add_argument("--dense-input-layers", type=int, default=4)
    p.add_argument("--bigmac-r", type=float, default=0.25)
    p.add_argument("--rope-theta", type=float, default=1_000_000.0)
    p.add_argument("--bos-token-id", type=int, default=2)
    p.add_argument("--eos-token-id", type=int, default=1)
    p.add_argument("--pad-token-id", type=int, default=0)
    args = p.parse_args()

    cfg = build_config(args)

    print("config:")
    for k in ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "head_dim",
              "max_seq_len", "tie_word_embeddings", "quasar_layers",
              "gated_layers", "num_routed_experts", "top_k",
              "shared_expert_size", "routed_expert_size", "bigmac_r",
              "dense_input_layers", "memory_slots", "memory_dim",
              "num_loops", "rope_theta"):
        print(f"  {k} = {getattr(cfg, k)}")

    with init_empty_weights():
        model = QuasarForCausalLM(cfg)

    total, active, by_class, by_class_active = count_params(model, cfg)

    print("\nparams (total / active per token):")
    for bucket in sorted(set(by_class) | set(by_class_active)):
        t = by_class.get(bucket, 0)
        a = by_class_active.get(bucket, 0)
        print(f"  {bucket:24s}  total {t/1e9:7.3f}B   active {a/1e9:7.3f}B")
    print(f"  {'-'*24}  -----------------")
    print(f"  {'TOTAL':24s}  total {total/1e9:7.3f}B   active {active/1e9:7.3f}B")

    bf16_total_gb = total * 2 / (1024 ** 3)
    bf16_active_gb = active * 2 / (1024 ** 3)
    print(f"\nbf16 weight sizes: total {bf16_total_gb:.1f}GB / active {bf16_active_gb:.1f}GB per copy")


if __name__ == "__main__":
    main()
