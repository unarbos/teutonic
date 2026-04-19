#!/usr/bin/env python3
import argparse
import json

import torch
from transformers import AutoModelForCausalLM

from train_lora_token_ids import TokenIdsDataset, Collator


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    ds = TokenIdsDataset(args.data, args.seq_len)
    collator = Collator()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        use_safetensors=True,
    )
    model.eval()

    losses = []
    for i in range(0, len(ds), args.batch_size):
        batch = collator([ds[j] for j in range(i, min(i + args.batch_size, len(ds)))])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss.detach().cpu()))

    mean_loss = sum(losses) / len(losses)
    print(json.dumps({"n_batches": len(losses), "mean_loss": mean_loss}, indent=2))


if __name__ == "__main__":
    main()
