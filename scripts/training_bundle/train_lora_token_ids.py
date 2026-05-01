#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


class TokenIdsDataset(Dataset):
    def __init__(self, path, seq_len):
        self.rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    ids = obj["input_ids"][:seq_len]
                    if len(ids) < seq_len:
                        continue
                    self.rows.append(ids)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids = self.rows[idx]
        x = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x),
            "labels": x.clone(),
        }


@dataclass
class Collator:
    def __call__(self, features):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", type=str, default=None,
                    help="comma-separated module name suffixes; defaults to a Quasar-aware set")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
    )
    model.config.use_cache = False

    # Quasar (Teutonic-XXIV) module names: attn uses q_proj/k_proj/v_proj/o_proj
    # like Qwen3, but FFN paths split into the dense SwiGLU (ffn.gate / ffn.up
    # / ffn.down) and BigMac MoE (shared_experts.{i}.{gate,up,down},
    # w_down_proj, w_up_proj). LoRA on `experts_w12` / `experts_w3` is not
    # supported — they are nn.Parameter blocks, not Linear layers. Override
    # via --lora-target-modules if you want a custom set.
    target_modules = args.lora_target_modules.split(",") if args.lora_target_modules else [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate", "up", "down",
        "w_down_proj", "w_up_proj",
    ]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = TokenIdsDataset(args.train_data, args.seq_len)
    val_ds = TokenIdsDataset(args.val_data, args.seq_len)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(),
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "best_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "best_adapter"))


if __name__ == "__main__":
    main()
