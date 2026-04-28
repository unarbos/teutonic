#!/usr/bin/env bash
# Wrapper invoked inside the tmux session on the GPU box.
# Reads HF_TOKEN from /root/teutonic-mining/.hf_token (chmod 600).
set -euo pipefail
cd /root/teutonic-mining
export HF_TOKEN="$(cat .hf_token)"
exec ./venv/bin/python -u train_challenger.py \
  --work /root/teutonic-mining/work \
  --bundle /root/teutonic-mining/bundle \
  --upload-repo "${UPLOAD_REPO:-unconst/Teutonic-VIII-h0}" \
  --report-out /root/teutonic-mining/work/verdict.json \
  --hf-token "$HF_TOKEN" \
  --n-shards 2 \
  --shard-start 0 \
  --eval-shard 5 \
  --n-eval 1500 \
  --n-score 3000 \
  --train-per-iter 3000 \
  --val-size 300 \
  --max-iters 3 \
  --target-mu 0.05 \
  --micro-batch 2 \
  --grad-accum 8 \
  --lr 2e-4 \
  --epochs 1 \
  --lora-r 16 \
  --lora-alpha 32 \
  --n-gpus 8
