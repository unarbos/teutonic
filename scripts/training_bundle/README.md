# Teutonic LoRA Starter Bundle

Practical starter bundle for Subnet 3 / Teutonic challenger training.

## What this bundle does

- Reads fixed-length token-id shard samples (`uint32`, seq_len=2048)
- Scores samples with the current king using next-token loss
- Computes lightweight quality heuristics (repetition, diversity)
- Buckets samples into `general`, `hard`, `suspicious`
- Builds train/val jsonl files of pretokenized records
- Trains a LoRA adapter with a custom causal-LM script (best path for token-id data)
- Provides an Axolotl config template if you want to adapt it later
- Evaluates checkpoints with the same core objective as subnet eval (avg next-token loss)

## Why a custom training script?

For this dataset, the cleanest path is to stay in token-id space and avoid decode/re-tokenize drift.
Axolotl is still useful as a reference/config layer, but fixed token arrays are simpler to train with
using a direct HuggingFace/PEFT script.

## Files

- `score_samples.py` — score token-id samples from shard(s)
- `build_curriculum.py` — build train/val jsonl from scored samples
- `train_lora_token_ids.py` — multi-GPU LoRA training on token-id jsonl
- `eval_token_loss.py` — held-out next-token loss eval
- `merge_lora.py` — merge LoRA adapter into a standalone model
- `axolotl_gemma3_lora.yml` — Axolotl-style reference config

## JSONL format

Each training record looks like:

```json
{"input_ids": [7221, 236761, 107, ...]}
```

All sequences should ideally already be length 2048.

## Suggested workflow

1. Score candidate samples:

```bash
python teutonic/scripts/training_bundle/score_samples.py \
  --model unconst/Teutonic-I \
  --shard /path/to/shard.npy \
  --output scored.jsonl \
  --max-samples 5000
```

2. Build curriculum:

```bash
python teutonic/scripts/training_bundle/build_curriculum.py \
  --scores scored.jsonl \
  --train-out train.jsonl \
  --val-out val.jsonl
```

3. Train LoRA:

```bash
torchrun --nproc_per_node=8 teutonic/scripts/training_bundle/train_lora_token_ids.py \
  --base-model unconst/Teutonic-I \
  --train-data train.jsonl \
  --val-data val.jsonl \
  --output-dir outputs/teutonic-lora \
  --seq-len 2048 \
  --micro-batch-size 2 \
  --grad-accum 8 \
  --learning-rate 2e-4 \
  --epochs 1 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

4. Evaluate merged or adapter checkpoints on held-out data.

5. Merge adapter:

```bash
python teutonic/scripts/training_bundle/merge_lora.py \
  --base-model unconst/Teutonic-I \
  --adapter-dir outputs/teutonic-lora/best_adapter \
  --output-dir outputs/teutonic-merged
```

6. Upload merged model to HuggingFace and submit with the existing miner/reveal flow.

## Notes

- Best objective match for the subnet is plain causal LM next-token loss.
- Keep training context at 2048 to match validator conditions.
- Choose checkpoints by held-out loss, not chat vibes.
